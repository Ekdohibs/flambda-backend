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

(* Four functions, each after the first calling the one before, in a functor
   that does not use its argument. Before Simplify each closure after the first
   holds the one before in a value slot, in the code of the functor, and the
   block of the functor's result holds the four: five values, each after the
   first made of the one before. Simplify makes all five symbols, and the code
   of the functor returns the symbol of the block. A search that lifts a value
   only once its parts are symbols has to lift five times in turn here. The
   functor [Make] of the standard library's [Map] has such a chain, a longer
   one. (Found and reduced by T4.)

   This unit was reduced from the functor [Make] of stdlib/map.ml of this
   repository (lines 77 to 551), by cutting that functor down, step by step, for
   as long as the validation stopped in the same way. What is kept of the
   library is the name [Make], the form of a functor whose result is a
   structure, and the order in which four functions of that functor call one
   another: [height] (lines 85 to 87), [create] (lines 89 to 91), which calls
   [height], [bal] (lines 95 to 123), which calls [create], and
   [add_min_binding] (lines 351 to 354), which calls [bal]. [a], [b], [c] and
   [d] stand where those four stood, each after the first calling the one
   before. They do not reproduce the names or the bodies of those four functions
   of the library, and no definition of one of the four is here unchanged. The
   parameter of the functor is not the library's. The operator [+] is declared
   as many units of this directory declare it; this comment names no file of the
   library for it. The line numbers are those of the file when this comment was
   written.

   The notice at the head of this file is lines 1 to 14 of stdlib/map.ml, as
   that file has them. *)

external ( + ) : int -> int -> int = "%addint"

module Make (X : sig end) = struct
  let a x = x + 1

  let b x = a x

  let c x = b x

  let d x = c x
end
