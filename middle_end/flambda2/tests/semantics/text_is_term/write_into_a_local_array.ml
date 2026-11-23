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

(* A write into an array of values that may be on the stack: the array is a
   parameter that the source marks [local_]. The write is an assignment that is
   marked local, and the text says so of it ([%array_set] with [values] and
   [value.lassign]); it wrote neither, and the write was read as an assignment
   into an array of the heap. So the text is the term.

   The declaration of [store] follows that of [unsafe_set] in stdlib/array.ml of
   this repository (its lines 37 to 39). It has the type that the file writes
   for the arguments and the result, and the primitive; its name is another, and
   it has neither the kind quantifier of line 37 nor the [@@ portable] of line
   38. Its second line is line 39 of the file, byte for byte; no whole
   declaration of the file is here as it stands. The function [first] is not of
   the library. The line numbers are those of the file when this comment was
   written.

   The notice at the head of this file is lines 1 to 14 of stdlib/array.ml, as
   that file has them. *)

external store : ('a array[@local_opt]) -> int -> 'a -> unit
  = "%array_unsafe_set"

let first (local_ a : string array) (x : string) = store a 0 x
