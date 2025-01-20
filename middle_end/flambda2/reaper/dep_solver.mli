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

type result

val pp_result : Format.formatter -> result -> unit

val fixpoint : Global_flow_graph.graph -> result

type 'a unboxed_fields =
  | Not_unboxed of 'a
  | Unboxed of 'a unboxed_fields Global_flow_graph.Field.Map.t

type assigned =
  Variable.t unboxed_fields Global_flow_graph.Field.Map.t

type changed_representation =
  | Block_representation of
      (int * Flambda_primitive.Block_access_kind.t) unboxed_fields Global_flow_graph.Field.Map.t * int
  | Closure_representation of
      Value_slot.t unboxed_fields Global_flow_graph.Field.Map.t * Function_slot.t

(*
type result =
  { uses : use_result;
    aliases : alias_result;
    dual_graph : Global_flow_graph.Dual.graph;
    unboxed_fields : assigned Code_id_or_name.Map.t;
    changed_representation :
      changed_representation
      Code_id_or_name.Map.t
  }
*)

val map_unboxed_fields : ('a -> 'b) -> 'a unboxed_fields -> 'b unboxed_fields

val get_unboxed_fields : result -> Code_id_or_name.t -> assigned option

val get_changed_representation : result -> Code_id_or_name.t -> changed_representation option

val has_use : result -> Code_id_or_name.t -> bool

val field_used :
  result -> Code_id_or_name.t -> Global_flow_graph.Field.t -> bool

(** Color of node when producing the graph as a .dot *)
val print_color : result -> Code_id_or_name.t -> string
