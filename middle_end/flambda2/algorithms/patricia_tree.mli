(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*           Mark Shinwell and Leo White, Jane Street Europe              *)
(*                                                                        *)
(*   Copyright 2015--2020 Jane Street Group LLC                           *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

type set

type +!'a map

module Make (_ : sig
  val print : Format.formatter -> int -> unit
end) : sig
  module Set : sig
    include Container_types.Set with type elt = int and type t = set

    (** [restrict_to_top_bits ~num_bits ~top_bits t] is the subset of [t] made
        of the elements whose [num_bits] most significant bits are [top_bits].
        Its cost only depends on [num_bits]. *)
    val restrict_to_top_bits : num_bits:int -> top_bits:int -> t -> t

    (** For testing; should always return [true] *)
    val valid : t -> bool
  end

  module Map : sig
    include
      Container_types.Map_plus_iterator
        with type key = int
         and type 'a t = 'a map
        with module Set = Set

    (** See [Set.restrict_to_top_bits]. *)
    val restrict_to_top_bits : num_bits:int -> top_bits:int -> 'a t -> 'a t

    (** For testing; should always return [true] *)
    val valid : _ t -> bool
  end
end
