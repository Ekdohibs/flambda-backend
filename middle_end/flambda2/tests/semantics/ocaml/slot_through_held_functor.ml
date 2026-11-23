(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*                       Jean-Christophe Filliâtre                        *)
(*                                                                        *)
(*   Copyright 2023 CNRS                                                  *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* The shape of slot_through_held.ml in a functor, as stdlib/pqueue.ml has it.
   [Ord] is the argument of the functor; [lt] captures it; [smaller] captures
   [lt] and [Ord]; [smallest] captures [smaller] alone. Where Simplify inlines
   [smaller] and then [lt] into the code of [smallest], the read of [Ord] in the
   copy of [lt] is a projection from the closure that [smaller] holds in a value
   slot, and the closure of [smaller] is itself read from a value slot of
   [smallest]. Simplify makes it the read of the value slot of [smaller] that
   holds [Ord] (FlambdaSlotsThroughSpec.read_slots_through). The functor is not
   applied: the unit's own run enters no code.

   [ORD], [Make], [lt] and [smaller] are a source that V3 wrote for the measure
   of the rewrite named above; by V3's record, what it read for the source is a
   file of the compiler, middle_end/flambda2/from_lambda/closure_conversion.ml
   ([close_one_function]). By V3's note of that source, [lt] is [lt] of
   stdlib/pqueue.ml of this repository (lines 94 and 95), the comparison of the
   argument of the functor made strict; V3 has no record of whether that file
   was open when it wrote the source. C3 added [smallest] and the attribute of
   [smaller] for this unit, when it looked into the validation of [pop_min] of
   that file (lines 113 to 121) on its term, with cuts of the file by function
   at hand: with [smallest] the three functions keep the shape of [lt],
   [sift_down] (lines 98 to 111) and [pop_min], where [sift_down] calls [lt] and
   the comparison of the argument and [pop_min] calls [sift_down]. The name [lt]
   and the form of its body are the library's. The first test of [smaller] has
   the form of line 104 of that file, where one of two is chosen by [lt]; its
   second test, for equality, and its name are not of that file. [smallest] is a
   name of that file too, where it is a variable of [sift_down] (line 101); the
   function of that name here and its body are not of that file. [ORD] has the
   form of the library's signatures of ordered types (stdlib/set.ml, lines 22 to
   26), and [Make] and [Ord] are the names that stdlib/set.ml gives its functor
   and the argument of it (line 77); the name [ORD] is of neither file. The
   library's [lt] reads two elements of an array by their indices, and its other
   two functions move elements; these compare and return their arguments. No
   line of those functions is here. The operators [<] and [=] are declared as
   many units of the tests declare them; this comment names no file of the
   library for them. The line numbers are those of the file when this comment
   was written.

   The notice at the head of this file is lines 1 to 13 of stdlib/pqueue.ml, as
   that file has them. *)

external ( < ) : int -> int -> bool = "%lessthan"

external ( = ) : int -> int -> bool = "%equal"

module type ORD = sig
  type t

  val compare : t -> t -> int
end

module Make (Ord : ORD) = struct
  let[@inline] lt a b = Ord.compare a b < 0

  let[@inline] smaller a b =
    if lt a b then a else if Ord.compare a b = 0 then a else b

  let smallest a b c = smaller (smaller a b) c
end
