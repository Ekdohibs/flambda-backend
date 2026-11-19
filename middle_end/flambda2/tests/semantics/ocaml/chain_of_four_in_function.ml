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

(* The control of chain_of_five_in_function.ml: three closures and the block,
   four values, each after the first made of the one before. (By T4.)

   This unit is chain_of_five_in_function.ml with its last closure taken away.
   It derives from that unit, and through it from chain_of_five_in_functor.ml,
   and was not made from the library apart from them. The unit
   chain_of_five_in_functor.ml was reduced from the functor [Make] of
   stdlib/map.ml of this repository (lines 77 to 551). What is left of the
   library here is the order in which three functions of that functor call one
   another: [height] (lines 85 to 87), [create] (lines 89 to 91), which calls
   [height], and [bal] (lines 95 to 123), which calls [create]. The enclosing
   function is named [make]. [a], [b] and [c] stand where those three stood,
   each after the first calling the one before. They do not reproduce the names
   or the bodies of those three functions of the library, and no definition of
   one of the three is here unchanged. The operator [+] is declared as many
   units of this directory declare it; this comment names no file of the library
   for it. The line numbers are those of the file when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/map.ml, as
   that file has them. *)

external ( + ) : int -> int -> int = "%addint"

let make () =
  let a x = x + 1 in
  let b x = a x in
  let c x = b x in
  a, b, c
