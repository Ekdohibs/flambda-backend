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

(* A unit of the shape of the library's Int32: a C function whose result is not
   a value (an unboxed 32-bit integer), called under a handler, and a C function
   that is given a boxed one. The conversion takes a call of a C function of
   every arity. The run of the unit calls neither.

   Where the items are from, in stdlib/ of this repository as it was when the
   unit was written: all four are adapted from stdlib/int32.ml, and none is a
   line of it as it stands: [format] and [to_string] from its lines 75 and 76,
   [of_string] from lines 78 and 79 and [of_string_opt] from lines 81 to 83,
   with the mode annotations and the attribute [@inline available] taken out and
   the names of the C functions changed to names that begin with c_.

   The notice at the head of this file is that of stdlib/int32.ml, as that file
   has it. *)

external of_string : string -> (int32[@unboxed])
  = "c_int32_of_string" "c_int32_of_string_unboxed"

external format : string -> int32 -> string = "c_int32_format"

let to_string n = format "%d" n

let of_string_opt s = try Some (of_string s) with Failure _ -> None
