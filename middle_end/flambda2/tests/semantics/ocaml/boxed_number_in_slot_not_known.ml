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

(* A control of boxed_number_in_slot.ml: the number is not known where the unit
   is compiled, so the constant cannot go into the code of the function.

   This unit was written as a variation of boxed_number_in_slot.ml of this
   directory: it is that unit with the constant behind [opaque]. Its definition
   of [add_limit] keeps the form that that unit has of the function
   [unsigned_to_int] of stdlib/int64.ml (lines 57 to 63) and of
   stdlib/nativeint.ml (lines 56 to 62) of this repository: a number that is
   boxed once by [of_int], inside the definition and before its [fun], and that
   the function reads. The comment of that unit says how the form came to it,
   what is not of the library, and what the three declarations of that unit
   match. [opaque] declares the primitive ["%opaque"] under a name and a type of
   this unit. The line numbers are those of the two files when this comment was
   written.

   The notice at the head of this file is lines 1 to 14 of stdlib/int64.ml, as
   that file has them; lines 1 to 14 of stdlib/nativeint.ml are the same
   bytes. *)

external of_int : int -> int64 = "%int64_of_int"

external to_int : int64 -> int = "%int64_to_int"

external add : int64 -> int64 -> int64 = "%int64_add"

external opaque : 'a -> 'a = "%opaque"

let add_limit =
  let limit = of_int (opaque 1000) in
  fun n -> to_int (add n limit)
