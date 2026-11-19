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

(* As chain_of_five_in_functor.ml, with no functor: a function that makes four
   closures, each after the first calling the one before, and returns them in a
   block. Simplify makes the four closures and the block symbols, and the code
   of the function returns the symbol of the block. (By T4.)

   This unit derives from chain_of_five_in_functor.ml and was not made from the
   library apart from it. A function is in the place of the functor: [make]
   makes the four functions as closures, and the tuple that it returns stands
   for the block of the functor's result. That unit was reduced from the functor
   [Make] of stdlib/map.ml of this repository (lines 77 to 551). What is left of
   the library here is the order in which four functions of that functor call
   one another: [height] (lines 85 to 87), [create] (lines 89 to 91), which
   calls [height], [bal] (lines 95 to 123), which calls [create], and
   [add_min_binding] (lines 351 to 354), which calls [bal]. The enclosing
   function is named [make]. [a], [b], [c] and [d] stand where those four stood,
   each after the first calling the one before. They do not reproduce the names
   or the bodies of those four functions of the library, and no definition of
   one of the four is here unchanged. The operator [+] is declared as many units
   of this directory declare it; this comment names no file of the library for
   it. The line numbers are those of the file when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/map.ml, as
   that file has them. *)

external ( + ) : int -> int -> int = "%addint"

let make () =
  let a x = x + 1 in
  let b x = a x in
  let c x = b x in
  let d x = c x in
  a, b, c, d
