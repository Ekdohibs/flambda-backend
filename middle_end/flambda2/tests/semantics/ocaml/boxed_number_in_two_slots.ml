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

(* As boxed_number_in_slot.ml, with two functions that read the one boxed
   number, each from its own closure. The number is not a field of the module.
   Simplify puts the constant in the code of both and binds no symbol to the
   boxed number.

   This unit was written as a variation of boxed_number_in_slot.ml of this
   directory, whose definition of [add_limit] has the form of the function
   [unsigned_to_int] of stdlib/int64.ml (lines 57 to 63) and of
   stdlib/nativeint.ml (lines 56 to 62) of this repository. The comment of that
   unit says how the form came to it, what is not of the library, and what its
   three declarations match; they are declarations of this unit too. Here that
   form is not kept: [limit] is bound at the level of the module, and
   [add_limit] and [sub_limit] are functions of the module that read it, where
   that unit and the library bind the number inside the definition, before its
   [fun]. What stays of it is a number that is boxed once by [of_int] and that a
   function reads. [sub] is declared with the name, the type and the primitive
   that line 24 of stdlib/int64.ml declares it with; that line has marks of
   locality and of portability, which the line here has not, and what the line
   here was written from is not recorded. The line numbers are those of the two
   files when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/int64.ml, as
   that file has them; lines 1 to 14 of stdlib/nativeint.ml are the same
   bytes. *)

external of_int : int -> int64 = "%int64_of_int"

external to_int : int64 -> int = "%int64_to_int"

external add : int64 -> int64 -> int64 = "%int64_add"

external sub : int64 -> int64 -> int64 = "%int64_sub"

let limit = of_int 1000

let add_limit n = to_int (add n limit)

let sub_limit n = to_int (sub n limit)
