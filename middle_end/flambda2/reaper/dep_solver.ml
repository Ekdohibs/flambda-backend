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

type dep = Global_flow_graph.Dep.t

module Field = Global_flow_graph.Field

module type Graph = sig
  type graph

  module Node : Container_types.S

  type edge

  val fold_nodes : graph -> (Node.t -> 'a -> 'a) -> 'a -> 'a

  val fold_edges : graph -> Node.t -> (edge -> 'a -> 'a) -> 'a -> 'a

  val target : edge -> Node.t

  type state

  type elt

  val top : elt

  val is_top : elt -> bool

  val is_bottom : elt -> bool

  val elt_deps : elt -> Node.Set.t

  val join : state -> elt -> elt -> elt

  val widen : state -> old:elt -> elt -> elt

  val less_equal : state -> elt -> elt -> bool

  val propagate : state -> Node.t -> elt -> edge -> elt

  val propagate_top : state -> edge -> bool

  val get : state -> Node.t -> elt

  val set : state -> Node.t -> elt -> unit
end

module Make_Fixpoint (G : Graph) = struct
  module Node = G.Node
  module SCC = Strongly_connected_components.Make (Node)

  module Make_SCC = struct
    let depset (graph : G.graph) (n : Node.t) : Node.Set.t =
      G.fold_edges graph n
        (fun edge acc -> Node.Set.add (G.target edge) acc)
        Node.Set.empty

    let complete_domain acc s =
      Node.Set.fold
        (fun x acc ->
          if Node.Map.mem x acc then acc else Node.Map.add x Node.Set.empty acc)
        s acc

    let from_graph (graph : G.graph) (state : G.state) : SCC.directed_graph =
      G.fold_nodes graph
        (fun n acc ->
          let deps = depset graph n in
          let acc = complete_domain acc deps in
          (* For nodes which are already as [top], the fixpoint is already
             reached. We can safely ignore the dependency and process these
             nodes at the beginning, cutting some cycles. *)
          let deps =
            Node.Set.filter
              (fun other -> not (G.is_top (G.get state other)))
              deps
          in
          Node.Map.add n deps acc)
        Node.Map.empty
  end

  let propagate_tops (graph : G.graph) (roots : Node.Set.t) (state : G.state) =
    let rec loop stack =
      match stack with
      | [] -> ()
      | n :: stack ->
        let stack =
          G.fold_edges graph n
            (fun dep stack ->
              if G.propagate_top state dep
              then
                let target = G.target dep in
                if G.is_top (G.get state target)
                then stack
                else (
                  G.set state n G.top;
                  target :: stack)
              else stack)
            stack
        in
        loop stack
    in
    let stack = Node.Set.elements roots in
    List.iter (fun n -> G.set state n G.top) stack;
    loop stack

  let fixpoint_component (graph : G.graph) (state : G.state)
      (component : SCC.component) =
    match component with
    | No_loop id ->
      let current_elt = G.get state id in
      if not (G.is_bottom current_elt)
      then
        G.fold_edges graph id
          (fun dep () ->
            let propagated = G.propagate state id current_elt dep in
            if not (G.is_bottom propagated)
            then
              let target = G.target dep in
              let old = G.get state target in
              G.set state target (G.join state old propagated))
          ()
    | Has_loop ids ->
      let q = Queue.create () in
      (* Invariants: [!q_s] contails the elements that may be pushed in [q],
         that is, the elements of [ids] that are not already in [q]. *)
      let in_loop = Node.Set.of_list ids in
      let q_s = ref Node.Set.empty in
      let to_recompute_deps = Hashtbl.create 17 in
      List.iter (fun id -> Queue.push id q) ids;
      let push n =
        if Node.Set.mem n !q_s
        then (
          Queue.push n q;
          q_s := Node.Set.remove n !q_s)
      in
      let propagate id =
        let current_elt = G.get state id in
        if not (G.is_bottom current_elt)
        then
          G.fold_edges graph id
            (fun dep () ->
              let propagated = G.propagate state id current_elt dep in
              if not (G.is_bottom propagated)
              then
                let target = G.target dep in
                let old = G.get state target in
                if Node.Set.mem target in_loop
                then (
                  let widened = G.widen state ~old propagated in
                  if not (G.less_equal state widened old)
                  then (
                    let new_deps = G.elt_deps widened in
                    Node.Set.iter
                      (fun elt_dep ->
                        Hashtbl.replace to_recompute_deps elt_dep
                          (Node.Set.add target
                             (Option.value ~default:Node.Set.empty
                                (Hashtbl.find_opt to_recompute_deps elt_dep))))
                      new_deps;
                    G.set state target widened;
                    push target;
                    match Hashtbl.find_opt to_recompute_deps target with
                    | None -> ()
                    | Some elt_deps -> Node.Set.iter push elt_deps))
                else G.set state target (G.join state old propagated))
            ()
      in
      while not (Queue.is_empty q) do
        let n = Queue.pop q in
        q_s := Node.Set.add n !q_s;
        propagate n
      done

  let fixpoint_topo (graph : G.graph) (roots : Node.Set.t) (state : G.state) =
    propagate_tops graph roots state;
    let components =
      SCC.connected_components_sorted_from_roots_to_leaf
        (Make_SCC.from_graph graph state)
    in
    Array.iter
      (fun component -> fixpoint_component graph state component)
      components

  let check_fixpoint (graph : G.graph) (roots : Node.Set.t) (state : G.state) =
    (* Checks that the given state is a post-fixpoint for propagation, and that
       all roots are set to [Top]. *)
    Node.Set.iter
      (fun root -> assert (G.less_equal state G.top (G.get state root)))
      roots;
    G.fold_nodes graph
      (fun node () ->
        G.fold_edges graph node
          (fun dep () ->
            assert (
              G.less_equal state
                (G.propagate state node (G.get state node) dep)
                (G.get state (G.target dep))))
          ())
      ()
end

type field_elt =
  | Field_top
  | Field_vals of Code_id_or_name.Set.t

(** Represents the part of a value that can be accessed *)
type elt =
  | Top  (** Value completely accessed *)
  | Fields of
      { fields : field_elt Field.Map.t;
        uses : Code_id_or_name.Set.t
      }
      (** Only the given fields are accessed, each field either being completely accessed for [Field_top]
      or corresponding to the union of all the elements corresponding to all the
      [Code_id_or_name.t] in the set for [Field_vals]. *)
  | Bottom  (** Value not accessed *)

let pp_field_elt ppf elt =
  match elt with
  | Field_top -> Format.pp_print_string ppf "⊤"
  | Field_vals s -> Code_id_or_name.Set.print ppf s

let pp_elt ppf elt =
  match elt with
  | Top -> Format.pp_print_string ppf "⊤"
  | Bottom -> Format.pp_print_string ppf "⊥"
  | Fields fields ->
    Format.fprintf ppf "{ fields: %a; uses : %a }"
      (Field.Map.print pp_field_elt)
      fields.fields Code_id_or_name.Set.print fields.uses

module Graph = struct
  type graph = Global_flow_graph.graph

  module Node = Code_id_or_name

  type edge = Global_flow_graph.Dep.t

  let fold_nodes graph f init =
    Hashtbl.fold
      (fun n _ acc -> f n acc)
      (Global_flow_graph.name_to_dep graph)
      init

  let fold_edges graph n f init =
    match Hashtbl.find_opt (Global_flow_graph.name_to_dep graph) n with
    | None -> init
    | Some deps -> Global_flow_graph.Dep.Set.fold f deps init

  let target (dep : dep) : Code_id_or_name.t =
    match dep with
    | Alias { target }
    | Accessor { target; _ }
    | Alias_if_def { target; _ }
    | Propagate { target; _ } ->
      Code_id_or_name.name target
    | Use { target } | Constructor { target; _ } -> target

  type nonrec elt = elt

  let less_equal_elt e1 e2 =
    match e1, e2 with
    | Bottom, _ | _, Top -> true
    | (Top | Fields _), Bottom | Top, Fields _ -> false
    | Fields f1, Fields f2 ->
      if f1.uses == f2.uses && f1.fields == f2.fields
      then true
      else
        Code_id_or_name.Set.subset f1.uses f2.uses
        &&
        let ok = ref true in
        ignore
          (Field.Map.merge
             (fun _ e1 e2 ->
               (match e1, e2 with
               | None, _ -> ()
               | Some _, None -> ok := false
               | _, Some Field_top -> ()
               | Some Field_top, _ -> ok := false
               | Some (Field_vals e1), Some (Field_vals e2) ->
                 if not (Code_id_or_name.Set.subset e1 e2) then ok := false);
               None)
             f1.fields f2.fields);
        !ok

  let elt_deps elt =
    match elt with
    | Bottom | Top -> Code_id_or_name.Set.empty
    | Fields f ->
      Field.Map.fold
        (fun _ v acc ->
          match v with
          | Field_top -> acc
          | Field_vals v -> Code_id_or_name.Set.union v acc)
        f.fields Code_id_or_name.Set.empty

  let join_elt e1 e2 =
    if e1 == e2
    then e1
    else
      match e1, e2 with
      | Bottom, e | e, Bottom -> e
      | Top, _ | _, Top -> Top
      | Fields f1, Fields f2 ->
        let fields =
          Field.Map.union
            (fun _ e1 e2 ->
              match e1, e2 with
              | Field_top, _ | _, Field_top -> Some Field_top
              | Field_vals e1, Field_vals e2 ->
                Some (Field_vals (Code_id_or_name.Set.union e1 e2)))
            f1.fields f2.fields
        in
        let uses = Code_id_or_name.Set.union f1.uses f2.uses in
        Fields { fields; uses }

  let make_field_elt uses (k : Code_id_or_name.t) =
    match Hashtbl.find_opt uses k with
    | Some Top -> Field_top
    | None | Some (Bottom | Fields _) ->
      Field_vals (Code_id_or_name.Set.singleton k)

  let propagate uses (k : Code_id_or_name.t) (elt : elt) (dep : dep) : elt =
    match elt with
    | Bottom -> Bottom
    | Top | Fields _ -> (
      match dep with
      | Alias _ -> elt
      | Use _ -> Top
      | Accessor { relation; target } ->
        let fields = Field.Map.singleton relation (make_field_elt uses k) in
        let uses =
          Code_id_or_name.Set.singleton (Code_id_or_name.name target)
        in
        Fields { fields; uses }
      | Constructor { relation; _ } -> (
        match elt with
        | Bottom -> assert false
        | Top -> Top
        | Fields fields -> (
          try
            let elems =
              match Field.Map.find_opt relation fields.fields with
              | None -> Code_id_or_name.Set.empty
              | Some Field_top -> raise Exit
              | Some (Field_vals s) -> s
            in
            Code_id_or_name.Set.fold
              (fun n acc ->
                join_elt acc
                  (match Hashtbl.find_opt uses n with
                  | None -> Bottom
                  | Some e -> e))
              elems Bottom
          with Exit -> Top))
      | Alias_if_def { if_defined; _ } -> (
        match Hashtbl.find_opt uses if_defined with
        | None | Some Bottom -> Bottom
        | Some (Fields _ | Top) -> elt)
      | Propagate { source; _ } -> (
        match Hashtbl.find_opt uses source with
        | None -> Bottom
        | Some elt -> elt))

  let propagate_top uses (dep : dep) : bool =
    match dep with
    | Alias _ -> true
    | Use _ -> true
    | Accessor _ -> false
    | Constructor _ -> true
    | Alias_if_def { if_defined; _ } -> (
      match Hashtbl.find_opt uses if_defined with
      | None | Some Bottom -> false
      | Some (Fields _ | Top) -> true)
    | Propagate { source; _ } -> (
      match Hashtbl.find_opt uses source with
      | None | Some (Bottom | Fields _) -> false
      | Some Top -> true)

  let top = Top

  let is_top = function Top -> true | Bottom | Fields _ -> false

  let is_bottom = function Bottom -> true | Top | Fields _ -> false

  let widen _ ~old:elt1 elt2 = join_elt elt1 elt2

  let join _ elt1 elt2 = join_elt elt1 elt2

  let less_equal _ elt1 elt2 = less_equal_elt elt1 elt2

  type state = (Code_id_or_name.t, elt) Hashtbl.t

  let get state n =
    match Hashtbl.find_opt state n with None -> Bottom | Some elt -> elt

  let set state n elt = Hashtbl.replace state n elt
end

module Solver = Make_Fixpoint (Graph)

module Dual_graph = struct
  include Global_flow_graph.Dual
  module Node = Code_id_or_name

  type field_elt =
    | Field_top
    | Field_vals of Code_id_or_name.Set.t

  type elt =
    | Top  (** Any value can flow to this variable *)
    | Block of
        { fields : field_elt Field.Map.t;
          sources : Code_id_or_name.Set.t
        }
        (** This value can be produced at any of those sources.
          Its value can be extracted from the fields of those field sources  *)
    | Bottom  (** No value can flow here *)

  let pp_field_elt ppf elt =
    match elt with
    | Field_top -> Format.pp_print_string ppf "⊤"
    | Field_vals s -> Code_id_or_name.Set.print ppf s

  let pp_elt ppf elt =
    match elt with
    | Top -> Format.pp_print_string ppf "⊤"
    | Bottom -> Format.pp_print_string ppf "⊥"
    | Block { fields; sources } ->
      Format.fprintf ppf "@[<hov 2>{@ sources: %a;@ fields: %a }@]"
        Code_id_or_name.Set.print sources
        (Field.Map.print pp_field_elt)
        fields

  let fold_nodes (graph : graph) f init =
    Code_id_or_name.Map.fold (fun n _ acc -> f n acc) graph init

  let fold_edges (type a) (graph : graph) (n : Node.t) (f : edge -> a -> a)
      (init : a) : a =
    match Code_id_or_name.Map.find_opt n graph with
    | None -> init
    | Some deps -> List.fold_left (Fun.flip f) init deps

  let target (dep : edge) : Code_id_or_name.t =
    match dep with
    | Alias { target } | Accessor { target; _ } | Constructor { target; _ } ->
      target

  let less_equal_elt (e1 : elt) (e2 : elt) =
    match e1, e2 with
    | Bottom, _ | _, Top -> true
    | (Top | Block _), Bottom | Top, Block _ -> false
    | Block f1, Block f2 ->
      if e1 == e2
      then true
      else
        Code_id_or_name.Set.subset f1.sources f2.sources
        &&
        let ok = ref true in
        ignore
          (Field.Map.merge
             (fun _ e1 e2 ->
               (match e1, e2 with
               | None, _ -> ()
               | Some _, None -> ok := false
               | _, Some Field_top -> ()
               | Some Field_top, _ -> ok := false
               | Some (Field_vals e1), Some (Field_vals e2) ->
                 if not (Code_id_or_name.Set.subset e1 e2) then ok := false);
               None)
             f1.fields f2.fields);
        !ok

  let elt_deps elt =
    match elt with
    | Bottom | Top -> Code_id_or_name.Set.empty
    | Block f ->
      Field.Map.fold
        (fun _ v acc ->
          match v with
          | Field_top -> acc
          | Field_vals v -> Code_id_or_name.Set.union v acc)
        f.fields Code_id_or_name.Set.empty

  let join_elt e1 e2 =
    if e1 == e2
    then e1
    else
      match e1, e2 with
      | Bottom, e | e, Bottom -> e
      | Top, _ | _, Top -> Top
      | Block f1, Block f2 ->
        let fields =
          Field.Map.union
            (fun _ e1 e2 ->
              match e1, e2 with
              | Field_top, _ | _, Field_top -> Some Field_top
              | Field_vals e1, Field_vals e2 ->
                Some (Field_vals (Code_id_or_name.Set.union e1 e2)))
            f1.fields f2.fields
        in
        let sources = Code_id_or_name.Set.union f1.sources f2.sources in
        Block { fields; sources }

  let make_field_elt sources (k : Code_id_or_name.t) =
    match Hashtbl.find_opt sources k with
    | Some Top -> Field_top
    | None | Some (Bottom | Block _) ->
      Field_vals (Code_id_or_name.Set.singleton k)

  let propagate sources (k : Code_id_or_name.t) (elt : elt) (dep : edge) : elt =
    match elt with
    | Bottom -> Bottom
    | Top | Block _ -> (
      match dep with
      | Alias _ -> elt
      | Constructor { relation; target } ->
        Block
          { fields = Field.Map.singleton relation (make_field_elt sources k);
            sources = Code_id_or_name.Set.singleton target
          }
      | Accessor { relation; _ } -> (
        match elt with
        | Bottom -> assert false
        | Top -> Top
        | Block { fields; _ } -> (
          try
            let elems =
              match Field.Map.find_opt relation fields with
              | None -> Code_id_or_name.Set.empty
              | Some Field_top -> raise Exit
              | Some (Field_vals s) -> s
            in
            Code_id_or_name.Set.fold
              (fun n acc ->
                join_elt acc
                  (match Hashtbl.find_opt sources n with
                  | None -> Bottom
                  | Some e -> e))
              elems Bottom
          with Exit -> Top)))

  let propagate_top _sources (dep : edge) : bool =
    match dep with
    | Alias _ -> true
    | Constructor _ -> false
    | Accessor _ -> true

  let top = Top

  let is_top = function Top -> true | Bottom | Block _ -> false

  let is_bottom = function Bottom -> true | Top | Block _ -> false

  let widen _ ~old:elt1 elt2 = join_elt elt1 elt2

  let join _ elt1 elt2 = join_elt elt1 elt2

  let less_equal _ elt1 elt2 = less_equal_elt elt1 elt2

  type state = (Code_id_or_name.t, elt) Hashtbl.t

  let get state n =
    match Hashtbl.find_opt state n with None -> Bottom | Some elt -> elt

  let set state n elt = Hashtbl.replace state n elt

  let build_dual (graph : Graph.graph) (solution : Graph.state) :
      graph * Code_id_or_name.Set.t =
    let add graph from to_ =
      Code_id_or_name.Map.update from
        (function None -> Some [to_] | Some l -> Some (to_ :: l))
        graph
    in
    (* top_roots is the initialization of the fixpoint. We can only consider top
       values as potential roots because we think that every constructor descend
       from one that has at least one top as its arguments. This is somewhat
       safe because atomic values are top (let x = 1: x would be top) and there
       are no purely cyclic values, for instance let rec x = x :: x would be
       initialized from external C functions (so Top). And a loop produced
       through a function cannot terminate (let rec loop () = (loop ()) :: (loop
       ())), hence the value is never produced. Note that this last case might
       actually be tricky (dead code still has to be compiled) *)
    let top_roots = ref Code_id_or_name.Set.empty in
    let graph =
      Hashtbl.fold
        (fun node (deps : Global_flow_graph.Dep.Set.t) acc ->
          Global_flow_graph.Dep.Set.fold
            (fun dep acc ->
              match dep with
              | Alias { target } ->
                add acc (Code_id_or_name.name target) (Alias { target = node })
              | Alias_if_def { if_defined; target } -> (
                match Hashtbl.find_opt solution if_defined with
                | None | Some Bottom -> acc
                | Some (Fields _ | Top) ->
                  add acc
                    (Code_id_or_name.name target)
                    (Alias { target = node }))
              | Propagate _ ->
                (* CR ncourant/pchambart: verify the invariant that this edge
                   should already be in the graph (or added later) by an
                   alias_if_def *)
                acc
              | Constructor { relation; target } ->
                add acc target (Constructor { relation; target = node })
              | Accessor { relation; target } ->
                add acc
                  (Code_id_or_name.name target)
                  (Accessor { relation; target = node })
              | Use _ ->
                top_roots := Code_id_or_name.Set.add node !top_roots;
                acc)
            deps acc)
        (Global_flow_graph.name_to_dep graph) Code_id_or_name.Map.empty
    in
    graph, !top_roots
end

module Alias_solver = Make_Fixpoint (Dual_graph)

type alias_result = Dual_graph.state

type 'a unboxed_fields =
  | Not_unboxed of 'a
  | Unboxed of 'a unboxed_fields Field.Map.t

let rec pp_unboxed_elt pp_unboxed ppf = function
  | Not_unboxed x -> pp_unboxed ppf x
  | Unboxed fields -> Field.Map.print (pp_unboxed_elt pp_unboxed) ppf fields

(* type repr = | Unboxed_fields of Variable.t unboxed_fields |
   Changed_representation of Field.t unboxed_fields *)

type assigned = Variable.t unboxed_fields Field.Map.t

type changed_representation =
  | Block_representation of
      (int * Flambda_primitive.Block_access_kind.t) unboxed_fields Field.Map.t * int
  | Closure_representation of
      Value_slot.t unboxed_fields Field.Map.t * Function_slot.t

let pp_changed_representation ff = function
  | Block_representation (fields, size) ->
      Format.fprintf ff
        "(fields %a) (size %d)"
        (Field.Map.print (pp_unboxed_elt (fun ff (field, _) -> Format.pp_print_int ff field))) fields size
  | Closure_representation (fields, function_slot) ->
      Format.fprintf ff
        "(fields %a) (function_slot %a)"
        (Field.Map.print (pp_unboxed_elt Value_slot.print)) fields
        Function_slot.print function_slot

type result =
  { uses : Graph.state;
    db : Datalog.database;
    aliases : alias_result;
    dual_graph : Dual_graph.graph;
    unboxed_fields : assigned Code_id_or_name.Map.t;
    (* CR: [(Field.t, Constant.t) Either.t unboxed_fields Code_id_or_name.Map.t]
       ? *)
    changed_representation :
      changed_representation Code_id_or_name.Map.t
  }

let pp_result ppf res =
  let elts = List.of_seq @@ Hashtbl.to_seq res.uses in
  let pp ppf l =
    let pp_sep ppf () = Format.fprintf ppf ",@ " in
    let pp ppf (name, elt) =
      Format.fprintf ppf "%a: %a" Code_id_or_name.print name pp_elt elt
    in
    Format.pp_print_list ~pp_sep pp ppf l
  in
  Format.fprintf ppf "@[<hov 2>{@ %a@ }@]" pp elts;
  Format.fprintf ppf "%a@." Datalog.print res.db

module Usages_rel = Datalog.Schema.Relation2 (Code_id_or_name) (Code_id_or_name)

let usages_rel = Datalog.create_relation ~name:"usages" Usages_rel.columns

let usages_rel v1 v2 = Datalog.atom usages_rel [v1; v2]

let with_usages = true

let datalog_schedule_usages =
  let open Datalog in
  let open Global_flow_graph in
  let not = Datalog.not in
  let ( let$ ) xs f = compile xs f in
  let ( ==> ) h c = where h (deduce c) in
  (* usages *)
  let usages_accessor_1 =
    let$ [to_; relation; base; _var] = ["to_"; "relation"; "base"; "_var"] in
    [not (used_pred base); usages_rel to_ _var; accessor_rel to_ relation base]
    ==> usages_rel base base
  in
  let usages_accessor_2 =
    let$ [to_; relation; base] = ["to_"; "relation"; "base"] in
    [not (used_pred base); used_pred to_; accessor_rel to_ relation base]
    ==> usages_rel base base
  in
  let usages_alias =
    let$ [to_; from; usage] = ["to_"; "from"; "usage"] in
    [ not (used_pred from);
      not (used_pred to_);
      usages_rel to_ usage;
      alias_rel to_ from ]
    ==> usages_rel from usage
  in
  (* propagate *)
  let alias_from_used_propagate =
    let$ [if_used; to_; from] = ["if_used"; "to_"; "from"] in
    [used_pred if_used; propagate_rel if_used to_ from] ==> alias_rel to_ from
  in
  let used_from_alias_used =
    let$ [to_; from] = ["to_"; "from"] in
    [alias_rel to_ from; used_pred to_] ==> used_pred from
  in
  (* accessor *)
  let used_fields_from_accessor_used_fields =
    let$ [to_; relation; base; _var] = ["to_"; "relation"; "base"; "_var"] in
    [ not (used_pred base);
      not (used_pred to_);
      not (used_fields_top_rel base relation);
      accessor_rel to_ relation base;
      usages_rel to_ _var ]
    ==> used_fields_rel base relation to_
  in
  let used_fields_from_accessor_used_fields_top =
    let$ [to_; relation; base] = ["to_"; "relation"; "base"] in
    [not (used_pred base); used_pred to_; accessor_rel to_ relation base]
    ==> used_fields_top_rel base relation
  in
  (* constructor *)
  let alias_from_accessed_constructor =
    let$ [base; base_use; relation; from; to_] =
      ["base"; "base_use"; "relation"; "from"; "to_"]
    in
    [ not (used_pred from);
      not (used_fields_top_rel base_use relation);
      not (used_pred base);
      constructor_rel base relation from;
      usages_rel base base_use;
      used_fields_rel base_use relation to_ ]
    ==> alias_rel to_ from
  in
  let used_from_accessed_constructor =
    let$ [base; base_use; relation; from] =
      ["base"; "base_use"; "relation"; "from"]
    in
    [ constructor_rel base relation from;
      not (used_pred base);
      usages_rel base base_use;
      used_fields_top_rel base_use relation ]
    ==> used_pred from
  in
  let used_from_constructor_used =
    let$ [base; relation; from] = ["base"; "relation"; "from"] in
    [used_pred base; constructor_rel base relation from] ==> used_pred from
  in
  (* use *)
  let used_from_use_1 =
    let$ [to_; from; _var] = ["to_"; "from"; "_var"] in
    [usages_rel to_ _var; use_rel to_ from] ==> used_pred from
  in
  let used_from_use_2 =
    let$ [to_; from] = ["to_"; "from"] in
    [used_pred to_; use_rel to_ from] ==> used_pred from
  in
  Datalog.Schedule.(
    fixpoint
      [ saturate
          [ alias_from_used_propagate;
            used_from_alias_used;
            used_from_constructor_used;
            used_from_use_1;
            used_from_use_2;
            used_from_accessed_constructor ];
        saturate
          [ alias_from_accessed_constructor;
            used_fields_from_accessor_used_fields;
            used_fields_from_accessor_used_fields_top;
            usages_accessor_1;
            usages_accessor_2;
            usages_alias ] ])

let query_uses =
  let open Datalog in
  let open! Global_flow_graph in
  compile ["X"] (fun [x] -> where [used_pred x] (yield [x]))

let query_used_field_top =
  let open Datalog in
  let open! Global_flow_graph in
  if with_usages
  then
    compile ["X"; "U"; "F"] (fun [x; u; f] ->
        where [usages_rel x u; used_fields_top_rel u f] (yield [x; f]))
  else
    compile ["X"; "F"] (fun [x; f] ->
        where [used_fields_top_rel x f] (yield [x; f]))

let query_used_field =
  let open Datalog in
  let open! Global_flow_graph in
  if with_usages
  then
    compile ["X"; "U"; "F"; "y"] (fun [x; u; f; y] ->
        where [usages_rel x u; used_fields_rel u f y] (yield [x; f; y]))
  else
    compile ["X"; "F"; "Y"] (fun [x; f; y] ->
        where [used_fields_rel x f y] (yield [x; f; y]))

let _db_to_uses db =
  (* Format.eprintf "%a@." Database.print_database db; *)
  let open Datalog in
  let open! Global_flow_graph in
  let h = Hashtbl.create 17 in
  Cursor.iter query_uses db ~f:(fun [u] -> Hashtbl.replace h u Top);
  Cursor.iter query_used_field_top db ~f:(fun [u; f] ->
      let f = Field.decode f in
      let[@local] ff fields =
        Hashtbl.replace h u (Fields { fields = (Field.Map.add f Field_top fields); uses = assert false })
      in
      match Hashtbl.find_opt h u with
      | Some Bottom -> assert false
      | Some Top -> ()
      | None -> ff Field.Map.empty
      | Some (Fields {fields = f; _}) -> ff f);
  Cursor.iter query_used_field db ~f:(fun [u; f; v] ->
      let[@local] ff fields =
        let f = Field.decode f in
        let v_top = Hashtbl.find_opt h v = Some Top in
        let fields =
          if v_top
          then Field.Map.add f Field_top fields
          else
            match Field.Map.find_opt f fields with
            | None ->
              Field.Map.add f
                (Field_vals (Code_id_or_name.Set.singleton v))
                fields
            | Some Field_top -> fields
            | Some (Field_vals w) ->
              Field.Map.add f (Field_vals (Code_id_or_name.Set.add v w)) fields
        in
        Hashtbl.replace h u (Fields {fields; uses = assert false})
      in
      match Hashtbl.find_opt h u with
      | Some Bottom -> assert false
      | Some Top -> ()
      | None -> ff Field.Map.empty
      | Some (Fields {fields = f; _}) -> ff f);
  h


let datalog_schedule_no_usages =
  let open Datalog in
  let open Global_flow_graph in
  let not = Datalog.not in
  let ( let$ ) xs f = compile xs f in
  let ( ==> ) h c = where h (deduce c) in
  (* propagate *)
  let alias_from_used_propagate =
    let$ [if_used; to_; from] = ["if_used"; "to_"; "from"] in
    [used_pred if_used; propagate_rel if_used to_ from] ==> alias_rel to_ from
  in
  (* alias *)
  let used_fields_from_used_fields_alias =
    let$ [to_; from; relation; used_as] =
      ["to_"; "from"; "relation"; "used_as"]
    in
    [ not (used_pred from);
      not (used_pred to_);
      not (used_fields_top_rel from relation);
      not (used_fields_top_rel to_ relation);
      alias_rel to_ from;
      used_fields_rel to_ relation used_as ]
    ==> used_fields_rel from relation used_as
  in
  let used_fields_top_from_used_fields_alias_top =
    let$ [to_; from; relation] = ["to_"; "from"; "relation"] in
    [ not (used_pred from);
      not (used_pred to_);
      alias_rel to_ from;
      used_fields_top_rel to_ relation ]
    ==> used_fields_top_rel from relation
  in
  let used_from_alias_used =
    let$ [to_; from] = ["to_"; "from"] in
    [alias_rel to_ from; used_pred to_] ==> used_pred from
  in
  (* accessor *)
  let used_fields_from_accessor_used =
    let$ [to_; relation; base] = ["to_"; "relation"; "base"] in
    [not (used_pred base); accessor_rel to_ relation base; used_pred to_]
    ==> used_fields_top_rel base relation
  in
  let used_fields_from_accessor_used_fields =
    let$ [to_; relation; base; _f; _x] =
      ["to_"; "relation"; "base"; "_f"; "_x"]
    in
    [ not (used_pred base);
      not (used_pred to_);
      not (used_fields_top_rel base relation);
      accessor_rel to_ relation base;
      used_fields_rel to_ _f _x ]
    ==> used_fields_rel base relation to_
  in
  let used_fields_from_accessor_used_fields_top =
    let$ [to_; relation; base; _f] = ["to_"; "relation"; "base"; "_f"] in
    [ not (used_pred base);
      not (used_pred to_);
      not (used_fields_top_rel base relation);
      accessor_rel to_ relation base;
      used_fields_top_rel to_ _f ]
    ==> used_fields_rel base relation to_
  in
  (* constructor *)
  let alias_from_used_fields_constructor =
    let$ [base; relation; from; used_as] =
      ["base"; "relation"; "from"; "used_as"]
    in
    [used_fields_rel base relation used_as; constructor_rel base relation from]
    ==> alias_rel used_as from
  in
  let used_from_constructor_field_used =
    let$ [base; relation; from] = ["base"; "relation"; "from"] in
    [used_fields_top_rel base relation; constructor_rel base relation from]
    ==> used_pred from
  in
  let used_from_constructor_used =
    let$ [base; relation; from] = ["base"; "relation"; "from"] in
    [used_pred base; constructor_rel base relation from] ==> used_pred from
  in
  (* use *)
  let used_from_used_use =
    let$ [to_; from] = ["to_"; "from"] in
    [used_pred to_; use_rel to_ from] ==> used_pred from
  in
  let used_from_used_fields_top_use =
    let$ [to_; from; _f] = ["to_"; "from"; "_f"] in
    [used_fields_top_rel to_ _f; use_rel to_ from] ==> used_pred from
  in
  let used_from_used_fields_use =
    let$ [to_; from; _f; _x] = ["to_"; "from"; "_f"; "_x"] in
    [used_fields_rel to_ _f _x; use_rel to_ from] ==> used_pred from
  in
  Datalog.Schedule.(
    fixpoint
      [ saturate
          [ alias_from_used_propagate;
            alias_from_used_fields_constructor;
            used_from_used_fields_use;
            used_from_used_fields_top_use;
            used_from_alias_used;
            used_from_constructor_used;
            used_from_constructor_field_used;
            used_from_used_use ];
        saturate
          [ used_fields_top_from_used_fields_alias_top;
            used_fields_from_accessor_used ];
        saturate
          [ used_fields_from_used_fields_alias;
            used_fields_from_accessor_used_fields_top;
            used_fields_from_accessor_used_fields ] ])

let datalog_schedule =
  if with_usages then datalog_schedule_usages else datalog_schedule_no_usages

let exists_with_parameters cursor params db =
  Datalog.Cursor.fold_with_parameters cursor params db ~init:false
    ~f:(fun [] _ -> true)

let mk_exists_query params existentials f =
  Datalog.(
    compile [] (fun [] ->
        with_parameters params (fun params ->
            foreach existentials (fun existentials ->
                where (f params existentials) (yield [])))))

let used_pred_query =
  let open! Global_flow_graph in
  mk_exists_query ["X"] [] (fun [x] [] -> [used_pred x])

let has_use_with_usages, field_used_with_usages =
  let open! Global_flow_graph in
  let usages_query =
    mk_exists_query ["X"] ["Y"] (fun [x] [y] -> [usages_rel x y])
  in
  let used_field_top_query =
    mk_exists_query ["X"; "F"] ["U"] (fun [x; f] [u] ->
        [usages_rel x u; used_fields_top_rel u f])
  in
  let used_field_query =
    mk_exists_query ["X"; "F"] ["U"; "V"] (fun [x; f] [u; v] ->
        [usages_rel x u; used_fields_rel u f v])
  in
  ( (fun db x ->
      exists_with_parameters used_pred_query [x] db
      || exists_with_parameters usages_query [x] db),
    fun db x field ->
      let field = Field.encode field in
      exists_with_parameters used_pred_query [x] db
      || exists_with_parameters used_field_top_query [x; field] db
      || exists_with_parameters used_field_query [x; field] db )

let has_use_without_usages, field_used_without_usages =
  let open! Global_flow_graph in
  let used_fields_top_any_query =
    mk_exists_query ["X"] ["F"] (fun [x] [f] -> [used_fields_top_rel x f])
  in
  let used_fields_any_query =
    mk_exists_query ["X"] ["F"; "Y"] (fun [x] [f; y] -> [used_fields_rel x f y])
  in
  let used_fields_top_query =
    mk_exists_query ["X"; "F"] [] (fun [x; f] [] -> [used_fields_top_rel x f])
  in
  let used_fields_query =
    mk_exists_query ["X"; "F"] ["Y"] (fun [x; f] [y] -> [used_fields_rel x f y])
  in
  ( (fun db x ->
      exists_with_parameters used_pred_query [x] db
      || exists_with_parameters used_fields_top_any_query [x] db
      || exists_with_parameters used_fields_any_query [x] db),
    fun db x field ->
      let field = Field.encode field in
      exists_with_parameters used_pred_query [x] db
      || exists_with_parameters used_fields_top_query [x; field] db
      || exists_with_parameters used_fields_query [x; field] db )

let has_use =
  if with_usages then has_use_with_usages else has_use_without_usages

let field_used =
  if with_usages then field_used_with_usages else field_used_without_usages

let print_color {db; _} v =
  if exists_with_parameters used_pred_query [v] db
  then "#a7a7a7"
  else if has_use db v
  then "#f1c40f"
  else "white"

let has_use uses v =
  let old_is_used = Hashtbl.mem uses.uses v in
  let new_is_used = has_use uses.db v in
  if old_is_used <> new_is_used
  then
    Misc.fatal_errorf "Different is_used on %a (old %b, new %b)@."
      Code_id_or_name.print v old_is_used new_is_used;
  new_is_used

let field_used uses v f =
  let new_is_used = field_used uses.db v f in
  let old_is_used =
    match Hashtbl.find_opt uses.uses v with
    | None -> false
    | Some Bottom -> false
    | Some Top -> true
    | Some (Fields { fields; _ }) -> Field.Map.mem f fields
  in
  if old_is_used <> new_is_used
  then
    Misc.fatal_errorf "Different field_used on %a %a (old %b, new %b)@."
      Code_id_or_name.print v Field.print f old_is_used new_is_used;
  new_is_used

let _pp_dual_result ppf (res : Dual_graph.state) =
  let elts = List.of_seq @@ Hashtbl.to_seq res in
  let pp ppf l =
    let pp_sep ppf () = Format.fprintf ppf ",@ " in
    let pp ppf (name, elt) =
      Format.fprintf ppf "%a: %a" Code_id_or_name.print name Dual_graph.pp_elt
        elt
    in
    Format.pp_print_list ~pp_sep pp ppf l
  in
  Format.fprintf ppf "@[<hov 2>{@ %a@ }@]" pp elts

type problematic_uses =
  | Cannot_unbox_due_to_uses
  | No_problem of { use_aliases : Code_id_or_name.Set.t }

let problematic_uses ~for_destructuring_value elt =
  match elt with
  | Top -> Cannot_unbox_due_to_uses
  | Bottom ->
    No_problem { use_aliases = Code_id_or_name.Set.empty }
    (* Cannot_unbox_due_to_uses *)
  | Fields { fields; uses } ->
    if for_destructuring_value
       && Field.Map.exists
            (fun (field : Field.t) _ ->
              match[@ocaml.warning "-4"] field with
              (* We probably cannot destructure a closure that is called
                 somewhere. *)
              | Code_of_closure | Apply _ -> true
              | _ -> false)
            fields
    then Cannot_unbox_due_to_uses
    else No_problem { use_aliases = uses }

let can_change_representation ~for_destructuring_value dual dual_graph graph allocation_id
    =
  (* The representation can be changed only if we can track its uses. And all
     the use sites can be changed. If a different value where to flow to that
     same use site, it would not be possible to change the representation.

     Note: This alias constraint is not a strict requirement, we might lighten
     that later.

     We also add the constraint that for a set of closures to have a changed
     representation, it must be true of all the closures defined in it,
     regardless of whether they are actually used. *)
  (* CR ncourant: we probably want to memoize this function *)
  let check_single ~for_destructuring_value allocation_id =
  let uses =
    match Hashtbl.find_opt graph allocation_id with
    | None -> Bottom
    | Some uses -> uses
  in
  match problematic_uses ~for_destructuring_value uses with
  | Cannot_unbox_due_to_uses -> false
  | No_problem { use_aliases } ->
    let alias_dominated_by_allocation_id alias =
      match (Hashtbl.find_opt dual alias : Dual_graph.elt option) with
      | None -> true
      | Some Bottom -> true
      | Some Top -> false
      | Some (Block { sources; _ }) ->
        Code_id_or_name.Set.equal
          (Code_id_or_name.Set.singleton allocation_id)
          sources
    in
    alias_dominated_by_allocation_id allocation_id
    && Code_id_or_name.Set.for_all alias_dominated_by_allocation_id use_aliases
  in
  check_single ~for_destructuring_value allocation_id &&
  List.for_all (fun (edge : Dual_graph.edge) ->
      match[@ocaml.warning "-4"] edge with
      | Constructor { target; relation = Function_slot _ } ->
          check_single ~for_destructuring_value:false target
      | _ -> true)
    (match Code_id_or_name.Map.find_opt allocation_id dual_graph with None -> [] | Some l -> l)


let map_from_allocation_points_to_dominated dual =
  let map = ref Code_id_or_name.Map.empty in
  Hashtbl.iter
    (fun id (elt : Dual_graph.elt) ->
      match elt with
      | Bottom | Top -> ()
      | Block { sources; _ } -> (
        (* Sources are only allocation points, so if sources is not a singleton,
           then id has no dominator that can be an allocation point. *)
        match Code_id_or_name.Set.get_singleton sources with
        | None -> ()
        | Some elt ->
          map
            := Code_id_or_name.Map.update elt
                 (function
                   | None -> Some (Code_id_or_name.Set.singleton id)
                   | Some set -> Some (Code_id_or_name.Set.add id set))
                 !map))
    dual;
  !map

let rec mapi_unboxed_fields (not_unboxed : 'a -> 'b -> 'c) (unboxed : Field.t
   -> 'a -> 'a) (acc : 'a) (uf : 'b unboxed_fields) : 'c unboxed_fields = match
   uf with | Not_unboxed x -> Not_unboxed (not_unboxed acc x) | Unboxed f ->
   Unboxed (Field.Map.mapi (fun field uf -> mapi_unboxed_fields not_unboxed
   unboxed (unboxed field acc) uf) f)

let map_unboxed_fields f uf = mapi_unboxed_fields (fun () x -> f x) (fun _ ()
   -> ()) () uf

let can_unbox dual dual_graph graph ~dominated_by_allocation_points
    allocation_id =
  (* CR ncourant: we probably want to memoize this function *)
  can_change_representation ~for_destructuring_value:true dual dual_graph graph
    allocation_id
  &&
  let aliases =
    match
      Code_id_or_name.Map.find_opt allocation_id dominated_by_allocation_points
    with
    | Some x -> x
    | None -> Code_id_or_name.Set.empty
  in
  Code_id_or_name.Set.for_all
    (fun alias ->
       let edges = match Code_id_or_name.Map.find_opt alias dual_graph with None -> [] | Some l -> l in
      List.for_all
        (fun (edge : Dual_graph.edge) ->
          match edge with
          | Alias _ | Accessor _ -> true
          | Constructor { target; relation } ->
            let target_uses =
              match Hashtbl.find_opt graph target with
              | None -> Bottom
              | Some t -> t
            in
            let relation_is_used =
              match target_uses with
              | Bottom -> false
              | Top -> true
              | Fields { fields; _ } -> Field.Map.mem relation fields
            in
            (not relation_is_used)
            ||
            let relation_prevents_unboxing =
              match relation with
              | Field.Block _ -> false
              | Field.Value_slot _ -> false
              | Field.Function_slot _ -> true (* TODO back to false *)
              | Field.Code_of_closure -> true
              | Field.Is_int -> true
              | Field.Get_tag -> true
              | Field.Apply (_, _) ->
                (* This requires unboxing returns of function, maybe later *)
                let unclear = true in
                unclear
            in
            (not relation_prevents_unboxing)
            && can_change_representation ~for_destructuring_value:false dual dual_graph
                 graph target)
        edges)
    aliases

let fixpoint (graph_new : Global_flow_graph.graph) =
  let result = Hashtbl.create 17 in
  let uses =
    Global_flow_graph.used graph_new
    |> Hashtbl.to_seq_keys |> List.of_seq |> Code_id_or_name.Set.of_list
  in
  Gc.full_major ();
  let t0 = Sys.time () in
  Solver.fixpoint_topo graph_new uses result;
  let t1 = Sys.time () in
  Gc.full_major ();
  let t1' = Sys.time () in
  let datalog = Global_flow_graph.to_datalog graph_new in
  let stats = Datalog.Schedule.create_stats () in
  let db = Datalog.Schedule.run ~stats datalog_schedule datalog in
  let t2 = Sys.time () in
  Format.eprintf "EXISTING: %f, DATALOG: %f, SPEEDUP: %f@." (t1 -. t0)
    (t2 -. t1')
    ((t1 -. t0) /. (t2 -. t1'));
  Format.eprintf "%a@." Datalog.Schedule.print_stats stats;
  (* let result2 = db_to_uses db in *)
  (* Format.eprintf "OLD:@.%a@.@.NEW:@.%a@.@." pp_result result pp_result
     result2; Format.eprintf "DB:@.%a@." Database.print_database db; *)
  (* Format.eprintf "OLD RESULT:@.%a@." pp_result result; Format.eprintf
     "NEW_RESULT:@.%a@." Database.print_database (Database.filter_database (fun
     relation -> List.mem (Database.relation_name relation) ["used";
     "used_fields"]) _db); *)
  Solver.check_fixpoint graph_new uses result;
  (* Hashtbl.iter
    (fun k v ->
      let v2 = Hashtbl.find result2 k in
      if not (Graph.less_equal_elt v v2 && Graph.less_equal_elt v2 v)
      then
        Misc.fatal_errorf "KEY %a OLD %a NEW %a@." Code_id_or_name.print k
          pp_elt v pp_elt v2)
    result;
  Hashtbl.iter
    (fun k _v ->
      let _v2 = Hashtbl.find result k in
      ())
    result2; *)
  let dual_graph, roots = Dual_graph.build_dual graph_new result in
  let aliases = Hashtbl.create 17 in
  Alias_solver.fixpoint_topo dual_graph roots aliases;
  Format.eprintf "@.SAUCISSE XXX@.@.@.";
  let dominated_by_allocation_points =
    map_from_allocation_points_to_dominated aliases
  in
  let allocation_point_dominator =
    Code_id_or_name.Map.fold
      (fun alloc_point dominated acc ->
        Code_id_or_name.Set.fold
          (fun dom acc -> Code_id_or_name.Map.add dom alloc_point acc)
          dominated acc)
      dominated_by_allocation_points Code_id_or_name.Map.empty
  in
  Hashtbl.iter
    (fun code_or_name elt ->
      if can_change_representation ~for_destructuring_value:true aliases dual_graph result
           code_or_name
      then
        let path =
          match
            Code_id_or_name.Map.find_opt code_or_name
              dominated_by_allocation_points
          with
          | Some x -> x
          | None -> Code_id_or_name.Set.empty
        in
        Format.eprintf "%a => %a@.%a@." Code_id_or_name.print code_or_name
          pp_elt elt Code_id_or_name.Set.print path)
    result;
  Format.eprintf "@.UNBOXABLE XXX@.@.@.";
  let assigned : assigned Code_id_or_name.Map.t ref = ref Code_id_or_name.Map.empty in
  let to_unbox =
    Hashtbl.fold
      (fun code_or_name _elt to_unbox ->
        if can_unbox aliases dual_graph result ~dominated_by_allocation_points
             code_or_name
        then Code_id_or_name.Set.add code_or_name to_unbox
        else to_unbox)
      result Code_id_or_name.Set.empty
  in
  let to_change_representation =
    Hashtbl.fold
      (fun code_or_name _elt to_change_representation ->
        if (not (Code_id_or_name.Set.mem code_or_name to_unbox))
           && can_change_representation ~for_destructuring_value:false aliases dual_graph
                result code_or_name
        then Code_id_or_name.Set.add code_or_name to_change_representation
        else to_change_representation)
      result Code_id_or_name.Set.empty
  in
  let has_to_be_unboxed code_or_name =
    match
      Code_id_or_name.Map.find_opt code_or_name allocation_point_dominator
    with
    | None -> false
    | Some alloc_point -> Code_id_or_name.Set.mem alloc_point to_unbox
  in
  Code_id_or_name.Set.iter
    (fun code_or_name ->
      Format.eprintf "%a@." Code_id_or_name.print code_or_name;
      let to_patch =
        match
          Code_id_or_name.Map.find_opt code_or_name
            dominated_by_allocation_points
        with
        | None -> Code_id_or_name.Set.empty
        | Some x -> x
      in
      Code_id_or_name.Set.iter
        (fun to_patch ->
          let rec unbox_elt elt name_prefix =
            match elt with
            | Top ->
              Misc.fatal_errorf "Trying to unbox Top uses when unboxing %a"
                Code_id_or_name.print to_patch
            | Bottom -> Field.Map.empty
            | Fields { fields; _ } ->
              Field.Map.mapi
                (fun field field_elt ->
                  let new_name =
                    Flambda_colours.without_colours ~f:(fun () ->
                        Format.asprintf "%s_field_%a" name_prefix Field.print
                          field)
                  in
                  let[@local] default () =
                    (* TODO let ghost for debugging *)
                    Not_unboxed (Variable.create new_name)
                  in
                  match field_elt with
                  | Field_top -> default ()
                  | Field_vals flow_to ->
                    if Code_id_or_name.Set.is_empty flow_to
                    then Misc.fatal_errorf "Empty set in [Field_vals]";
                    if Code_id_or_name.Set.for_all has_to_be_unboxed flow_to
                    then
                      let elt =
                        Code_id_or_name.Set.fold
                          (fun flow acc ->
                            match Hashtbl.find_opt result flow with
                            | None ->
                              Misc.fatal_errorf
                                "%a is in [Field_vals] but not in result"
                                Code_id_or_name.print flow
                            | Some elt -> Graph.join_elt acc elt)
                          flow_to Bottom
                      in
                      Unboxed (unbox_elt elt new_name)
                    else if Code_id_or_name.Set.exists has_to_be_unboxed flow_to
                    then
                      Misc.fatal_errorf
                        "Field %a of %s flows to both unboxed and non-unboxed \
                         variables"
                        Field.print field name_prefix
                    else default ())
                fields
          in
          let new_name =
            Flambda_colours.without_colours ~f:(fun () ->
                Format.asprintf "%a_into_%a" Code_id_or_name.print code_or_name
                  Code_id_or_name.print to_patch)
          in
          let fields =
            match Hashtbl.find_opt result to_patch with
            | None -> Field.Map.empty
            | Some elt -> unbox_elt elt new_name
          in
          assigned := Code_id_or_name.Map.add to_patch fields !assigned)
        to_patch)
    to_unbox;
  Format.printf "new vars: %a"
    (Code_id_or_name.Map.print
       (Field.Map.print (pp_unboxed_elt Variable.print)))
    !assigned;
  let changed_representation = ref Code_id_or_name.Map.empty in
  Code_id_or_name.Set.iter
    (fun code_id_or_name ->
      let uses =
        match Hashtbl.find_opt result code_id_or_name with
        | None -> Bottom
        | Some x -> x
      in
      let r = ref ~-1 in
      let mk_field () =
        incr r;
        (!r, 
         Flambda_primitive.(Block_access_kind.Values { tag = Unknown; size = Unknown; field_kind = Block_access_field_kind.Any_value }))
      in
      let repr =
        let rec repr_elt mk_field = function
          | Top ->
            Misc.fatal_errorf "Cannot change representation of Top for %a"
              Code_id_or_name.print code_id_or_name
          | Bottom -> Field.Map.empty
          | Fields { fields; _ } ->
            (* TODO handle closures & non-value fields *)
            Field.Map.filter_map
              (fun field field_elt ->
                 match field with
                 | Code_of_closure | Apply _ -> None
                 | Get_tag | Is_int | Block _ | Value_slot _ | Function_slot _ ->
                Some (match field_elt with
                | Field_top -> Not_unboxed (mk_field ())
                | Field_vals flow_to ->
                  if Code_id_or_name.Set.is_empty flow_to
                  then Misc.fatal_errorf "Empty set in [Field_vals]";
                  if Code_id_or_name.Set.for_all has_to_be_unboxed flow_to
                  then
                    let elt =
                      Code_id_or_name.Set.fold
                        (fun flow acc ->
                          match Hashtbl.find_opt result flow with
                          | None ->
                            Misc.fatal_errorf
                              "%a is in [Field_vals] but not in result"
                              Code_id_or_name.print flow
                          | Some elt -> Graph.join_elt acc elt)
                        flow_to Bottom
                    in
                    Unboxed (repr_elt mk_field elt)
                  else if Code_id_or_name.Set.exists has_to_be_unboxed flow_to
                  then
                    Misc.fatal_errorf
                      "Field %a of %a flows to both unboxed and non-unboxed \
                       variables"
                      Field.print field Code_id_or_name.print code_id_or_name
                  else Not_unboxed (mk_field ())))
              fields
        in
        if match uses with
          | Bottom -> true | Top -> assert false
          | Fields { fields; _ } ->
              not (Field.Map.exists (fun field _ -> match field with Block _ | Is_int | Get_tag -> false | Code_of_closure | Apply _ | Value_slot _ | Function_slot _ -> true) fields)
        then
        let repr = repr_elt mk_field uses in
        Block_representation (repr, !r + 1)
        else
          let mk_field () =
            Value_slot.create (Compilation_unit.get_current_exn ())
              ~name:"unboxed_value_slot" Flambda_kind.With_subkind.any_value (* TODO *) 
          in
          let repr = repr_elt mk_field uses in
          Closure_representation (repr,
                                  Function_slot.create (Compilation_unit.get_current_exn ()) ~name:"unboxed_function_slot" Flambda_kind.With_subkind.any_value)
      in
      Code_id_or_name.Set.iter
        (fun c ->
          changed_representation
            := Code_id_or_name.Map.add c repr !changed_representation)
        (match Code_id_or_name.Map.find_opt code_id_or_name dominated_by_allocation_points with None -> Code_id_or_name.Set.empty (*XXX check this*) | Some s -> s))
    to_change_representation;
  Format.eprintf "@.TO_CHG: %a@."
    (Code_id_or_name.Map.print pp_changed_representation)
    !changed_representation;
  { uses = result;
    db;
    aliases;
    dual_graph;
    (* unboxed_fields = !assigned;
    changed_representation = !changed_representation *)
    unboxed_fields = Code_id_or_name.Map.empty ;
    changed_representation = Code_id_or_name.Map.empty
  }

let get_unboxed_fields uses cn =
  Code_id_or_name.Map.find_opt cn uses.unboxed_fields

let get_changed_representation uses cn =
  Code_id_or_name.Map.find_opt cn uses.changed_representation
