(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*                        Guillaume Bury, OCamlPro                        *)
(*                                                                        *)
(*   Copyright 2023--2023 OCamlPro SAS                                    *)
(*   Copyright 2023--2023 Jane Street Group LLC                           *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* This is mainly to have indexable unique ids *)
module Id : sig
  type t
  val fresh : unit -> t
  val print : Format.formatter -> t -> unit
  module Map : Container_types.Map with type key = t
end = struct
  type t = int
  let print = Numbers.Int.print
  let fresh =
    let r = ref 0 in
    (fun () -> incr r; !r)
  module Tree = Patricia_tree.Make(struct let print = print end)
  module Map = Tree.Map
end

type t = {
  new_params_indexed: Bound_parameter.t Id.Map.t;
}

let print ppf { new_params_indexed; } =
  Format.fprintf ppf "@[<hov 1>(\
      @[<hov 1>(new_params_indexed@ %a)@]\
      )@]"
    (Id.Map.print Bound_parameter.print) new_params_indexed

let empty =
  { new_params_indexed = Id.Map.empty; }

let is_empty { new_params_indexed; } =
  Id.Map.is_empty new_params_indexed

let new_param t bound_param =
  (* create a fres var/bound_param to index the new parameter *)
  let id = Id.fresh () in
  let new_params_indexed = Id.Map.add id bound_param t.new_params_indexed in
  { (* t with *) new_params_indexed;  }

let rename t =
  let bindings = Id.Map.bindings t.new_params_indexed in
  let keys, bound_param_list = List.split bindings in
  let bound_params = Bound_parameters.create bound_param_list in
  let new_bound_params = Bound_parameters.rename bound_params in
  let renaming = Bound_parameters.renaming bound_params ~guaranteed_fresh:new_bound_params in
  let new_params_indexed = Id.Map.of_list (List.combine keys (Bound_parameters.to_list new_bound_params)) in
  { (* t with *) new_params_indexed; }, renaming

let fold ~init ~f { new_params_indexed; } =
  Id.Map.fold f new_params_indexed init

let rec find_arg id = function
  | [] ->
    Misc.fatal_errorf
      "Missing lifted param id: %a not found in lifted_cont_params stack"
      Id.print id
  | { new_params_indexed; } :: r ->
    match Id.Map.find_opt id new_params_indexed with
    | Some param -> Bound_parameter.simple param
    | None -> find_arg id r

let args ~callee_lifted_params ~caller_stack_lifted_params =
  fold callee_lifted_params ~init:[]
    ~f:(fun id _callee_param acc -> find_arg id caller_stack_lifted_params :: acc)

let bound_parameters t =
  Bound_parameters.create @@
  fold t ~init:[]
    ~f:(fun _id bound_param acc -> bound_param :: acc)
