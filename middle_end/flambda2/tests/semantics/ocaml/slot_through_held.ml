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

(* [compare] is a parameter of [make]. [lt] captures it; [smaller] captures [lt]
   and [compare]; [smallest] captures [smaller] alone. Where Simplify inlines
   [smaller] and then [lt] into the code of [smallest], the read of [compare] in
   the copy of [lt] is a projection from the closure that [smaller] holds in a
   value slot, and the closure of [smaller] is itself read from a value slot of
   [smallest]. Simplify makes it the read of the value slot of [smaller] that
   holds [compare]. The closure that is read from is not [my_closure] here: it
   is one that the body reads from a value slot and knows the code of
   (FlambdaSlotsThroughSpec.read_slots_through). It is the shape of [lt],
   [sift_down] and [pop_min] in stdlib/pqueue.ml. [make] is called through
   [opaque], so that the unit's own run enters the code of [smallest].

   This unit is C3's variation of slot_through_held_functor.ml of this directory
   outside a functor: [compare] is a parameter of [make] where it is the member
   of the argument of the functor there, and [lt], [smaller] and [smallest] are
   the functions of that unit. The comment of that unit says how it was made and
   what is and what is not of stdlib/pqueue.ml of this repository. [difference]
   and [result] are C3's, of this unit alone. The operators [<], [=] and [-] and
   [opaque] are declared as many units of the tests declare them; this comment
   names no file of the library for them.

   The notice at the head of this file is lines 1 to 13 of stdlib/pqueue.ml, as
   that file has them. *)

external ( < ) : int -> int -> bool = "%lessthan"

external ( = ) : int -> int -> bool = "%equal"

external ( - ) : int -> int -> int = "%subint"

external opaque : 'a -> 'a = "%opaque"

let make (compare : int -> int -> int) =
  let[@inline] lt a b = compare a b < 0 in
  let[@inline] smaller a b =
    if lt a b then a else if compare a b = 0 then a else b
  in
  let smallest a b c = smaller (smaller a b) c in
  smallest

let difference a b = a - b

let result = (opaque make) difference 3 1 2
