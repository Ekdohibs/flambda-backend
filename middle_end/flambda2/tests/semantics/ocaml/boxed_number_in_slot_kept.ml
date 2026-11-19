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

(* A control of boxed_number_in_slot.ml: the boxed number is a field of the
   module too, so Simplify binds a symbol to it, and that symbol is the one that
   stands for it.

   This unit was written as a variation of boxed_number_in_slot.ml of this
   directory, whose definition of [add_limit] has the form of the function
   [unsigned_to_int] of stdlib/int64.ml (lines 57 to 63) and of
   stdlib/nativeint.ml (lines 56 to 62) of this repository. The comment of that
   unit says how the form came to it, what is not of the library, and what its
   three declarations match; they are the declarations of this unit. Here that
   form is not kept: [limit] is bound at the level of the module, and
   [add_limit] is a function of the module that reads it, where that unit and
   the library bind the number inside the definition, before its [fun]. What
   stays of it is a number that is boxed once by [of_int] and that a function
   reads. The line numbers are those of the two files when this comment was
   written.

   The notice at the head of this file is lines 1 to 14 of stdlib/int64.ml, as
   that file has them; lines 1 to 14 of stdlib/nativeint.ml are the same
   bytes. *)

external of_int : int -> int64 = "%int64_of_int"

external to_int : int64 -> int = "%int64_to_int"

external add : int64 -> int64 -> int64 = "%int64_add"

let limit = of_int 1000

let add_limit n = to_int (add n limit)
