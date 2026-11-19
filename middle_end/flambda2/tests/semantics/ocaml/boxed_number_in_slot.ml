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

(* A boxed number that a closure holds in a value slot. The unit makes the
   number once, of a constant, and the function adds it to its argument.
   Simplify puts the constant in the code of the function, whose closure then
   has no value slot, and binds no symbol to the boxed number, which nothing
   reads any more.

   The definition of [add_limit] has the form of the function [unsigned_to_int]
   of stdlib/int64.ml (lines 57 to 63) and of stdlib/nativeint.ml (lines 56 to
   62) of this repository: a number that is boxed once by [of_int], inside the
   definition and before its [fun], and that the function reads. The form came
   to this unit through a diagnosis, by another stream of this work, of why
   those two files were not validated: it named that function and quoted how it
   binds its number before its [fun]. Whether either file was also read for this
   unit is not recorded. The function of the library tests that its argument is
   in a range and returns an option; the name, the constant, the body and the
   result of [add_limit] are not the library's, and no line of that function is
   here. [of_int], [to_int] and [add] are declared with the names, the types and
   the primitives that lines 34, 35 and 23 of stdlib/int64.ml declare them with;
   those lines have marks of locality and of portability, which the lines here
   have not, and what the lines here were written from is not recorded. The line
   numbers are those of the two files when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/int64.ml, as
   that file has them; lines 1 to 14 of stdlib/nativeint.ml are the same
   bytes. *)

external of_int : int -> int64 = "%int64_of_int"

external to_int : int64 -> int = "%int64_to_int"

external add : int64 -> int64 -> int64 = "%int64_add"

let add_limit =
  let limit = of_int 1000 in
  fun n -> to_int (add n limit)
