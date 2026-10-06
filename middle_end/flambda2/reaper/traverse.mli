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

module For_types : sig
  type ('f, 'a) t =
    | With_types : 'a -> ([`With_types], 'a) t
    | No_types : ([`No_types], 'a) t

  val map : ('a -> 'b) -> ('f, 'a) t -> ('f, 'b) t
end

module Problem : sig
  type 't t =
    { deps : Global_flow_graph.graph;
      code_deps : Traverse_acc.code_dep Code_id.Map.t;
      delayed_deps : Traverse_acc.delayed_deps;
      applications : Traverse_acc.Applications.t;
      free_names : Name_occurrences.t;
      all_sets_of_closures :
        ( 't,
          (Name.t * Code_id.t Or_unknown.t) Function_slot.Lmap.t list )
        For_types.t;
      final_typing_env : ('t, typing_env option) For_types.t;
      module_symbol : ('t, Symbol.t) For_types.t
    }
end

module Skeleton : sig
  type t =
    { toplevel_expr : Rev_expr.t;
      code : Rev_expr.rev_code Code_id.Map.t;
      ordered_code_ids : Code_id.t array;
      fixed_arity_continuations : Continuation.Set.t;
      continuation_info : Traverse_acc.continuation_info Continuation.Map.t
    }
end

val run :
  Flambda_unit.t ->
  final_typing_env:('f, typing_env option) For_types.t ->
  free_names:Name_occurrences.t ->
  'f Problem.t * Skeleton.t
