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

let bound_parameters t =
  Bound_parameters.create (Id.Map.data t.new_params_indexed)

let fold ~init ~f { new_params_indexed; } =
  Id.Map.fold f new_params_indexed init

let find_arg id { new_params_indexed; } =
  match Id.Map.find_opt id new_params_indexed with
  | Some param -> Bound_parameter.simple param
  | None ->
    Misc.fatal_errorf "Missing lifted param id in lifted_cont_params"

let args ~callee_lifted_params ~caller_lifted_params =
  let map =
    Id.Map.merge (fun _ callee_param_opt caller_param_opt ->
        match callee_param_opt, caller_param_opt with
        | None, None -> assert false (* invariant of the Map module *)
        | Some _, None ->
          Misc.fatal_errorf "Missing lifted param"
        | None, Some _ -> None
        | Some _, Some bp -> Some (Bound_parameter.simple bp)
      ) callee_lifted_params.new_params_indexed
      caller_lifted_params.new_params_indexed
  in
  Id.Map.data map

