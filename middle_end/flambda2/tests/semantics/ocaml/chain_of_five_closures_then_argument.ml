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

(* As chain_of_five_in_functor.ml, with a chain that the block does not end:
   five functions, each after the first calling the one before, and a sixth that
   calls the fifth and uses the functor's argument. Before Simplify each closure
   after the first holds the one before in a value slot, in the code of the
   functor. Simplify makes the five closures symbols; the closure of the sixth,
   which holds the argument, and the block of the functor's result stay in the
   code of the functor. So the five values that are lifted in turn are all
   closures, where in chain_of_five_in_functor.ml the fifth is the block. (By
   T4.)

   This unit derives from chain_of_five_in_functor.ml and was not made from the
   library apart from it: the chain has one function more, and a last function
   uses the parameter of the functor. That unit was reduced from the functor
   [Make] of stdlib/map.ml of this repository (lines 77 to 551). What is kept of
   the library here is the name [Make], the form of a functor whose result is a
   structure, and the order in which four functions of that functor call one
   another: [height] (lines 85 to 87), [create] (lines 89 to 91), which calls
   [height], [bal] (lines 95 to 123), which calls [create], and
   [add_min_binding] (lines 351 to 354), which calls [bal]. [a], [b], [c] and
   [d] stand where those four stood, each after the first calling the one
   before; [e] stands for no function of the library. The last function, [f],
   uses the parameter of the functor. Functions of the library's functor use
   theirs by calling [Ord.compare], and in the unit as it was first written [f]
   called [Ord.compare] of a parameter of the library's signature [OrderedType];
   for this directory the parameter became a signature with one value, [v], and
   [f] adds [X.v]. The parameter of the functor is not the library's. The
   functions of this unit do not reproduce the names or the bodies of those four
   functions of the library, and no definition of one of the four is here
   unchanged. The operator [+] is declared as many units of this directory
   declare it; this comment names no file of the library for it. The line
   numbers are those of the file when this comment was written.

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

  let e x = d x

  let f x = e x + X.v
end
