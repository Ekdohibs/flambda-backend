(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* Seventeen functions in the body of a functor, each after the first calling
   the one before, none using the argument of the functor. Simplify makes the
   seventeen closures and the block of the result of the functor symbols. A
   lifting of the validation takes only a value whose parts are symbols or
   constants already, so the validation would lift them one after the other,
   eighteen liftings in turn, which is more than the test tool tries: the unit
   is here for the line that says so.

   From the declaration of the operator [+] on, this unit is another unit of
   this directory, chain_of_five_in_functor.ml, with thirteen functions added
   before its [end]: [e] to [q], each calling the one before. That unit was
   reduced from the functor [Make] of stdlib/map.ml of this repository (lines 77
   to 551); its comment says how. What that unit keeps of the library is here in
   the lines that are that unit's: the name [Make], the form of a functor whose
   result is a structure, and, in [a], [b], [c] and [d], the order in which four
   functions of that functor call one another: [height] (lines 85 to 87),
   [create] (lines 89 to 91), [bal] (lines 95 to 123) and [add_min_binding]
   (lines 351 to 354). The thirteen functions added stand for no function of the
   library: they are there to make the chain longer than the test tool tries.
   The functions [a] to [q] do not reproduce the names or the bodies of those
   four functions of the library, and no definition of one of the four is here
   unchanged. The parameter of the functor is not the library's. The operator
   [+] is declared as many units of this directory declare it; this comment
   names no file of the library for it. The line numbers are those of the file
   when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/map.ml, as
   that file has them. *)

external ( + ) : int -> int -> int = "%addint"

module Make (X : sig end) = struct
  let a x = x + 1

  let b x = a x

  let c x = b x

  let d x = c x

  let e x = d x

  let f x = e x

  let g x = f x

  let h x = g x

  let i x = h x

  let j x = i x

  let k x = j x

  let l x = k x

  let m x = l x

  let n x = m x

  let o x = n x

  let p x = o x

  let q x = p x
end
