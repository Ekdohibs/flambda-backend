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

(* The control of chain_of_five_closures_then_argument.ml: four functions, each
   after the first calling the one before, and a fifth that calls the fourth and
   uses the argument of the functor. Simplify makes the four closures symbols.
   (By T4.)

   This unit is chain_of_five_closures_then_argument.ml with one function fewer
   in the chain. It derives from that unit, and through it from
   chain_of_five_in_functor.ml, and was not made from the library apart from
   them. The unit chain_of_five_in_functor.ml was reduced from the functor
   [Make] of stdlib/map.ml of this repository (lines 77 to 551). What is kept of
   the library here is the name [Make], the form of a functor whose result is a
   structure, and the order in which four functions of that functor call one
   another: [height] (lines 85 to 87), [create] (lines 89 to 91), which calls
   [height], [bal] (lines 95 to 123), which calls [create], and
   [add_min_binding] (lines 351 to 354), which calls [bal]. [a], [b], [c] and
   [d] stand where those four stood, each after the first calling the one
   before. The last function, [e], uses the parameter of the functor. Functions
   of the library's functor use theirs by calling [Ord.compare], and in the unit
   as it was first written [e] called [Ord.compare] of a parameter of the
   library's signature [OrderedType]; for this directory the parameter became a
   signature with one value, [v], and [e] adds [X.v]. The parameter of the
   functor is not the library's. The functions of this unit do not reproduce the
   names or the bodies of those four functions of the library, and no definition
   of one of the four is here unchanged. The operator [+] is declared as many
   units of this directory declare it; this comment names no file of the library
   for it. The line numbers are those of the file when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/map.ml, as
   that file has them. *)

external ( + ) : int -> int -> int = "%addint"

module Make (X : sig
  val v : int
end) =
struct
  let a x = x + 1

  let b x = a x

  let c x = b x

  let d x = c x

  let e x = d x + X.v
end
