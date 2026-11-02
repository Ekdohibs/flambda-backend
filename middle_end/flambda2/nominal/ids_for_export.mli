(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*                       Vincent Laviron, OCamlPro                        *)
(*                                                                        *)
(*   Copyright 2020 OCamlPro SAS                                          *)
(*   Copyright 2020 Jane Street Group LLC                                 *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

module Simple = Int_ids.Simple

type t

val empty : t

val create :
  ?symbols:Symbol.Set.t ->
  ?variables:Variable.Set.t ->
  ?simples:Simple.Set.t ->
  ?consts:Reg_width_const.Set.t ->
  ?code_ids:Code_id.Set.t ->
  ?continuations:Continuation.Set.t ->
  ?fields:Field.Set.t ->
  unit ->
  t

val singleton_variable : Variable.t -> t

val singleton_code_id : Code_id.t -> t

val singleton_continuation : Continuation.t -> t

val singleton_symbol : Symbol.t -> t

val singleton_field : Field.t -> t

val from_simple : Simple.t -> t

val add_variable : t -> Variable.t -> t

val add_symbol : t -> Symbol.t -> t

val add_name : t -> Name.t -> t

val add_const : t -> Reg_width_const.t -> t

val add_simple : t -> Simple.t -> t

val add_code_id : t -> Code_id.t -> t

val add_code_id_or_name : t -> Code_id_or_name.t -> t

val add_continuation : t -> Continuation.t -> t

val add_field : t -> Field.t -> t

val union : t -> t -> t

val union_list : t list -> t

(** The set must only contain identifiers built using [Table_by_int_id]. *)
val of_int_set : Patricia_tree.set -> t

val symbols : t -> Symbol.Set.t

val variables : t -> Variable.Set.t

val simples : t -> Simple.Set.t

val consts : t -> Reg_width_const.Set.t

val code_ids : t -> Code_id.Set.t

val continuations : t -> Continuation.Set.t

val fields : t -> Field.Set.t
