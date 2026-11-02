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

module Coercion = Int_ids.Coercion
module Simple = Int_ids.Simple
module Flags = Int_ids.Flags

(* All identifiers are integers whose top bits identify their kind (see
   [Int_ids.Flags]), and all their sets are Patricia trees, so we store them in
   a single set. *)
module Tree = Patricia_tree.Make (struct
  let print = Format.pp_print_int
end)

module Set = Tree.Set

type t = Set.t

let of_symbols (symbols : Symbol.Set.t) : t = Obj.magic symbols

let of_variables (variables : Variable.Set.t) : t = Obj.magic variables

let of_simples (simples : Simple.Set.t) : t = Obj.magic simples

let of_consts (consts : Reg_width_const.Set.t) : t = Obj.magic consts

let of_code_ids (code_ids : Code_id.Set.t) : t = Obj.magic code_ids

let of_continuations (continuations : Continuation.Set.t) : t =
  Obj.magic continuations

let of_fields (fields : Field.Set.t) : t = Obj.magic fields

let field_id (field : Field.t) : int = Obj.magic field

let only_flags flags t =
  Set.restrict_to_top_bits ~num_bits:Table_by_int_id.Id.flags_size_in_bits
    ~top_bits:flags t

let of_int_set (set : Patricia_tree.set) : t = set

let symbols t : Symbol.Set.t = Obj.magic (only_flags Flags.symbol t)

let variables t : Variable.Set.t = Obj.magic (only_flags Flags.variable t)

let simples t : Simple.Set.t = Obj.magic (only_flags Flags.simple t)

let consts t : Reg_width_const.Set.t = Obj.magic (only_flags Flags.const t)

let code_ids t : Code_id.Set.t = Obj.magic (only_flags Flags.code_id t)

let continuations t : Continuation.Set.t =
  Obj.magic (only_flags Flags.continuation t)

let fields t : Field.Set.t = Obj.magic (only_flags Flags.field t)

let empty = Set.empty

let union = Set.union

let create ?(symbols = Symbol.Set.empty) ?(variables = Variable.Set.empty)
    ?(simples = Simple.Set.empty) ?(consts = Reg_width_const.Set.empty)
    ?(code_ids = Code_id.Set.empty) ?(continuations = Continuation.Set.empty)
    ?(fields = Field.Set.empty) () =
  union (of_symbols symbols)
    (union (of_variables variables)
       (union (of_simples simples)
          (union (of_consts consts)
             (union (of_code_ids code_ids)
                (union (of_continuations continuations) (of_fields fields))))))

let singleton_variable (var : Variable.t) = Set.singleton (var :> int)

let singleton_code_id (code_id : Code_id.t) = Set.singleton (code_id :> int)

let singleton_continuation (cont : Continuation.t) = Set.singleton (cont :> int)

let singleton_symbol (symbol : Symbol.t) = Set.singleton (symbol :> int)

let singleton_field field = Set.singleton (field_id field)

let add_const t (const : Reg_width_const.t) = Set.add (const :> int) t

let add_variable t (var : Variable.t) = Set.add (var :> int) t

let add_symbol t (sym : Symbol.t) = Set.add (sym :> int) t

let add_name t (name : Name.t) = Set.add (name :> int) t

let add_simple t (simple : Simple.t) =
  let t =
    if Coercion.is_id (Simple.coercion simple)
    then t
    else Set.add (simple :> int) t
  in
  Simple.pattern_match simple ~const:(add_const t)
    ~name:(fun name ~coercion:_ -> add_name t name)

let add_code_id t (code_id : Code_id.t) = Set.add (code_id :> int) t

let add_code_id_or_name t (code_id_or_name : Code_id_or_name.t) =
  Set.add (code_id_or_name :> int) t

let add_continuation t (continuation : Continuation.t) =
  Set.add (continuation :> int) t

let add_field t field = Set.add (field_id field) t

let from_simple simple = add_simple empty simple

let rec union_list ts =
  match ts with [] -> empty | t :: ts -> union t (union_list ts)
